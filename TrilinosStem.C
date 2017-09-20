#include "TrilinosStem.h"

#include "BelosSolverFactory.hpp"
#include <BelosTpetraAdapter.hpp>

// Teuchos
#include "Teuchos_RCP.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_GlobalMPISession.hpp"
#include "Teuchos_DefaultComm.hpp"

#include <Ifpack2_Factory.hpp>

// MueLu main header: include most common header files in one line
//#include <MueLu.hpp>
#include <MueLu_TpetraOperator.hpp>
#include <MueLu_TpetraOperator_def.hpp>
//#include <MueLu_Hierarchy_decl.hpp>
//#include <MueLu_Hierarchy_def.hpp>
//#include <MueLu_TpetraOperator_decl.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
//#include <MueLu_Utilities.hpp>
//#include <MueLu_UseShortNames.hpp>

//#include <Epetra_CrsMatrix.h>
//#include <ml_MultiLevelPreconditioner.h>
//#include <Xpetra_EpetraCrsMatrix.hpp>

// Belos
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosBlockCGSolMgr.hpp"
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosXpetraAdapter.hpp" // this header defines Belos::XpetraOp()
#include "BelosMueLuAdapter.hpp"  // this header defines Belos::MueLuOp()
#include "BelosOperatorT.hpp"

// Xpetra
#include <Xpetra_Map.hpp>
#include <Xpetra_MapFactory.hpp>
#include <Xpetra_CrsMatrixWrap.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_Parameters.hpp>

#include <Xpetra_Matrix.hpp>
#include <MueLu_CoordinatesTransferFactory.hpp>
#include <Xpetra_TpetraMultiVector.hpp>

#define ARRAY3D(i,j,k,imin,jmin,kmin,ni,nj) (i-(imin))+(((j)-(jmin))*(ni))+(((k)-(kmin))*(ni*nj))


Teuchos::RCP<const TrilinosStem::Map> TrilinosStem::map;
Teuchos::RCP<const TrilinosStem::XMap> TrilinosStem::xmap;
Teuchos::RCP<TrilinosStem::XMultiVector> TrilinosStem::xcoordinates;
Teuchos::RCP<TrilinosStem::Matrix> TrilinosStem::A;
Teuchos::RCP<TrilinosStem::Vector> TrilinosStem::b;
Teuchos::RCP<TrilinosStem::Vector> TrilinosStem::x;
int* TrilinosStem::myGlobalIndices_;
int TrilinosStem::numLocalElements_;
int TrilinosStem::MyPID;

Teuchos::RCP<Teuchos::ParameterList> TrilinosStem::solverParams;
Teuchos::RCP<Belos::LinearProblem<double, TrilinosStem::MultiVector, TrilinosStem::Operator> > TrilinosStem::problem;
Teuchos::RCP<Belos::SolverManager<double, TrilinosStem::MultiVector, TrilinosStem::Operator> > TrilinosStem::solver;
//Teuchos::RCP<Ifpack2::Preconditioner<TrilinosStem::Scalar, TrilinosStem::Ordinal, TrilinosStem::Ordinal, TrilinosStem::Node> > TrilinosStem::preconditioner;

extern "C" {
    void setup_trilinos_(
            int* nx,
            int* ny,
	    int* nz,
            int* localNx,
            int* localNy,
	    int* localNz,
            int* local_xmin,
            int* local_xmax,
            int* local_ymin,
            int* local_ymax,
	    int* local_zmin,
            int* local_zmax,
	    double* dx,
	    double* dy,
	    double* dz,
            int* tl_max_iters,
            double* tl_eps)
    {
        TrilinosStem::initialise(*nx, *ny, *nz,
                *localNx, *localNy, *localNz,
                *local_xmin,
                *local_xmax,
                *local_ymin,
                *local_ymax,
		*local_zmin,
		*local_zmax,
		*dx,
		*dy,
		*dz,
                *tl_max_iters,
                *tl_eps);
    }

    void trilinos_solve_(
            int* nx,
            int* ny,
	    int* nz,
            int* local_xmin,
            int* local_xmax,
            int* local_ymin,
            int* local_ymax,
	    int* local_zmin,
            int* local_zmax,
            int* global_xmin,
            int* global_xmax,
            int* global_ymin,
            int* global_ymax,
	    int* global_zmin,
            int* global_zmax,
            int* numIters,
            double* rx,
            double* ry,
	    double* rz,
            double* Kx,
            double* Ky,
	    double* Kz,
            double* u0)
    {
        TrilinosStem::solve(
                *nx,
                *ny,
		*nz,
                *local_xmin,
                *local_xmax,
                *local_ymin,
                *local_ymax,
		*local_zmin,
                *local_zmax,
                *global_xmin,
                *global_xmax,
                *global_ymin,
                *global_ymax,
		*global_zmin,
                *global_zmax,
                numIters,
                *rx,
                *ry,
		*rz,
                Kx,
                Ky,
		Kz,
                u0);
    }

    void finalise_trilinos_()
    {
        TrilinosStem::finalise();
    }
}

void TrilinosStem::initialise(
        int nx,
        int ny,
	int nz,
        int localNx,
        int localNy,
	int localNz,
        int local_xmin,
        int local_xmax,
        int local_ymin,
        int local_ymax,
	int local_zmin,
        int local_zmax,
	double dx,
	double dy,
	double dz,
        int tl_max_iters,
        double tl_eps)
{
    Platform &platform = Tpetra::DefaultPlatform::getDefaultPlatform();
    Teuchos::RCP<const Teuchos::Comm<int> > comm = platform.getComm();
    MyPID = comm->getRank();

    if(MyPID == 0)
    {
        std::cout << "[STEM]: Setting up Trilinos/Tpetra...";
    }

    int numGlobalElements = nx * ny * nz;
    numLocalElements_ = localNx * localNy * localNz;
//    int rowSpacing = nx - localNx; not needed

    myGlobalIndices_ = new int[numLocalElements_];

    int n = 0;

    {
        int index = local_xmin;
        int i = 0;
        for(int l = local_zmin; l <= local_zmax; l++) {
            for(int k = local_ymin; k <= local_ymax; k++) {
                for(int j = local_xmin; j <= local_xmax; j++) {
	        myGlobalIndices_[i] = ARRAY3D(j,k,l,1,1,1,nx,ny)+1;
                i++;
                index++;
               }
            }
	}
    }

    if(MyPID == 0)
    {
        std::cerr << "\t[STEM]: creating map...";
    }
    Teuchos::ArrayView<const Ordinal> localElementList = Teuchos::ArrayView<const Ordinal>(myGlobalIndices_, numLocalElements_);
    Teuchos::RCP<Node> node = platform.getNode();
    map = Teuchos::rcp(new Map(numGlobalElements, localElementList, 1, comm, node));

    //Store co-ordinates of the unknowns so we can use brick agglomeration
    //need to use xmap rather tha map as MueLu needs an Xpetra::MultiVector object

    xmap = Xpetra::MapFactory<Ordinal,Ordinal,Node>::Build(Xpetra::UnderlyingLib::UseTpetra,numGlobalElements, localElementList, 1, comm, node);
    xcoordinates = Xpetra::MultiVectorFactory<Scalar,Ordinal,Ordinal,Node>::Build(xmap,3);
    Teuchos::ArrayRCP<Teuchos::ArrayRCP<Scalar> > Coord(3);
    Coord[0] = xcoordinates->getDataNonConst(0);
    Coord[1] = xcoordinates->getDataNonConst(1);
    Coord[2] = xcoordinates->getDataNonConst(2);
    {
        int i = 0;
        for(int l = local_zmin; l <= local_zmax; l++) {
            for(int k = local_ymin; k <= local_ymax; k++) {
                for(int j = local_xmin; j <= local_xmax; j++) {
	            Coord[0][i] = (Teuchos::as<Scalar>(j)-1.5)*dx;
	            Coord[1][i] = (Teuchos::as<Scalar>(k)-1.5)*dy;
		    Coord[2][i] = (Teuchos::as<Scalar>(l)-1.5)*dz;
                    i++;
               }
           }
	}
    }

    if(MyPID == 0)
    {
        std::cerr << " DONE. " << std::endl;
    }

    size_t* numNonZero = new size_t[numLocalElements_];

    int i = 0;
    for(int l = local_zmin; l <= local_zmax; l++) {
        for(int k = local_ymin; k <= local_ymax; k++) {
            for(int j = local_xmin; j <= local_xmax; j++) {
                size_t nnz = 1;

                if(1 != k)
                    nnz++;
                if(ny != k)
                    nnz++;
                if(1 != j)
                    nnz++;
                if(nx != j)
                    nnz++;
                if(1 != l)
                    nnz++;
                if(nz != l)
                    nnz++;

                numNonZero[i] = nnz;
                i++;
            }
        }
    }

    if(MyPID == 0)
    {
        std::cerr << "\t[STEM]: creating CrsMatrix...";
    }
    Teuchos::ArrayRCP<const size_t> nnz = Teuchos::ArrayRCP<const size_t>(numNonZero, 0, numLocalElements_, false);

    A = Teuchos::rcp(new Matrix(map, nnz, Tpetra::StaticProfile));
    
    if(MyPID == 0)
    {
        std::cerr << " DONE." << std::endl;
    }

    b = Teuchos::rcp(new Vector(map));
    x = Teuchos::rcp(new Vector(map));

    solverParams = Teuchos::parameterList();

    // Belos+MueLu parameters - create sublists
    Teuchos::ParameterList& iterationParams      = (*solverParams).sublist("Belos Parameters",false,"Belos sublist");
    Teuchos::ParameterList& preconditionerParams = (*solverParams).sublist("MueLu Parameters",false,"MueLu sublist");

    // Hardwired defaults
    iterationParams.set("Maximum Iterations", tl_max_iters);
    iterationParams.set("Convergence Tolerance", tl_eps); 
    int verbosity  = Belos::Errors;
    verbosity     += Belos::Warnings;
    verbosity     += Belos::TimingDetails;
    verbosity     += Belos::StatusTestDetails;
    verbosity     += Belos::IterationDetails;
    verbosity     += Belos::FinalSummary;
    //verbosity     += Belos::Debug;
    iterationParams.set("Verbosity", verbosity);
    //iterationParams.set("Verbosity", Belos::Errors);
    //iterationParams.set("Output Frequency", 1); // disabling turns off the residual history
    iterationParams.set("Output Style", (int) Belos::Brief);

    // Override with parameters set in the XML file (should test it exists)
    std::string OptionsFile = "Options.xml";
    Teuchos::updateParametersFromXmlFileAndBroadcast(OptionsFile, Teuchos::inoutArg(*solverParams),*comm,true);
}

void TrilinosStem::solve(
        int nx,
        int ny,
	int nz,
        int local_xmin,
        int local_xmax,
        int local_ymin,
        int local_ymax,
	int local_zmin,
        int local_zmax,
        int global_xmin,
        int global_xmax,
        int global_ymin,
        int global_ymax,
	int global_zmin,
        int global_zmax,
        int* numIters,
        double rx,
        double ry,
	double rz,
        double* Kx,
        double* Ky,
	double* Kz,
        double* u0)
{
    Belos::SolverFactory<Scalar, MultiVector, Operator> factory;

    // Belos+MueLu parameters - retrieve sublists
    Teuchos::ParameterList& iterationParams      = (*solverParams).sublist("Belos Parameters" ,false,"Belos sublist");
    Teuchos::ParameterList& preconditionerParams = (*solverParams).sublist("MueLu Parameters" ,false,"MueLu sublist");
    Teuchos::ParameterList& tealeafParams        = (*solverParams).sublist("Solver Parameters",false,"Solver sublist");

    //set defaults to CG with no preconditioner
    std::string belos_name     = "CG";
    bool preconditioner_active = false;
    if(tealeafParams.isParameter("Belos Solver"))
    {
        belos_name            = Teuchos::getParameter<std::string>(tealeafParams,"Belos Solver");
    }
    if(tealeafParams.isParameter("MueLu Preconditioner"))
    {
        preconditioner_active = Teuchos::getParameter<bool>(tealeafParams,"MueLu Preconditioner");
    }

    //solver = factory.create("RCG", belosParams);
    //solver = factory.create("GMRES", belosParams);
    solver = factory.create(belos_name, Teuchos::rcpFromRef(iterationParams));

    Teuchos::RCP<Teuchos::FancyOStream> fos = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));

    if(MyPID == 0)
    {
        std::cout << std::endl << "Solver parameters set:" << std::endl;
        tealeafParams.print();
        std::cout << std::endl << "Belos solver description:" << std::endl;
        (*solver).describe(*fos,Teuchos::VERB_EXTREME);
        std::cout << std::endl << "Belos solver parameters available:" << std::endl;
        solver->getValidParameters()->print();
        std::cout << std::endl << "Belos solver parameters set:" << std::endl;
        solver->getCurrentParameters()->print();
	if(preconditioner_active)
	{
            std::cout << std::endl << "MueLu preconditioner parameters available:" << std::endl;
            Teuchos::RCP< const Teuchos::ParameterList > validParams = MueLu::MasterList::List();
            validParams->print();
            std::cout << std::endl << "MueLu preconditioner parameters set:" << std::endl;
            preconditionerParams.print();
	}
	std::cout << std::endl;
    }

    bool insertValues = false;

    if(A->isFillComplete()) {
        A->resumeFill();
    } else {
        insertValues = true;
    }

    //A->setAllToScalar(0.0);

    //std::vector<double> Values(4);
    //std::vector<int> Indices(4);
    std::vector<double> Values(6);
    std::vector<int> Indices(6);

    int numEntries = 0;
    //int local_nx = local_xmax - local_xmin + 1 + 4;
    int local_nx = local_xmax - local_xmin + 1 + 4;
    int local_ny = local_ymax - local_ymin + 1 + 4;
    
    int i = 0;
    
    for(int l = local_zmin; l <= local_zmax; l++) {
        for(int k = local_ymin; k <= local_ymax; k++) {
            for(int j = local_xmin; j <= local_xmax; j++) {
	      
                Values.clear();
                Indices.clear();

                double c2 = -rx*Kx[ARRAY3D(j,k,l,local_xmin-2,local_ymin-2,local_zmin-2,local_nx,local_ny)];
                double c3 = -rx*Kx[ARRAY3D(j+1,k,l,local_xmin-2,local_ymin-2,local_zmin-2,local_nx,local_ny)];
                double c4 = -ry*Ky[ARRAY3D(j,k,l,local_xmin-2,local_ymin-2,local_zmin-2,local_nx,local_ny)];
                double c5 = -ry*Ky[ARRAY3D(j,k+1,l,local_xmin-2,local_ymin-2,local_zmin-2,local_nx,local_ny)];
                double c6 = -rz*Kz[ARRAY3D(j,k,l,local_xmin-2,local_ymin-2,local_zmin-2,local_nx,local_ny)];
                double c7 = -rz*Kz[ARRAY3D(j,k,l+1,local_xmin-2,local_ymin-2,local_zmin-2,local_nx,local_ny)];		

                numEntries = 0;

		// Entries in x
                if(1 != j) {
                    numEntries++;
                    Values.push_back(c2);
                    Indices.push_back(myGlobalIndices_[i]-1);
                }


                if(nx != j) {
                    numEntries++;
                    Values.push_back(c3);
                    Indices.push_back(myGlobalIndices_[i]+1);
                }
                // Entries in y
                if(1 != k) {
                    numEntries++;
                    Values.push_back(c4);
                    Indices.push_back(myGlobalIndices_[i]-nx);
                }

                if(ny != k) {
                    numEntries++;
                    Values.push_back(c5);
                    Indices.push_back(myGlobalIndices_[i]+nx);
                }
                // Entres in z
                if(1 != l) {
                    numEntries++;
                    Values.push_back(c6);
                    Indices.push_back(myGlobalIndices_[i]-nx*ny);
                }

                if(ny != l) {
                    numEntries++;
                    Values.push_back(c7);
                    Indices.push_back(myGlobalIndices_[i]+nx*ny);
                }                
                

                double diagonal = (1.0 -c2 -c3 -c4 -c5 - c6 - c7);

                if (insertValues) {
                    A->insertGlobalValues(myGlobalIndices_[i],
                            Teuchos::ArrayView<Ordinal>(&Indices[0], numEntries), 
                            Teuchos::ArrayView<Scalar>(&Values[0], numEntries));

                    A->insertGlobalValues(myGlobalIndices_[i],
                            Teuchos::tuple<Ordinal>( myGlobalIndices_[i] ),
                            Teuchos::tuple<Scalar>(diagonal));
                } else {
                    A->replaceGlobalValues(myGlobalIndices_[i],
                            Teuchos::ArrayView<Ordinal>(&Indices[0], numEntries),
                            Teuchos::ArrayView<Scalar>(&Values[0], numEntries));

                    A->replaceGlobalValues(myGlobalIndices_[i],
                            Teuchos::tuple<Ordinal>( myGlobalIndices_[i] ),
                            Teuchos::tuple<Scalar>(diagonal));
                }

                i++;
            }
        }
    }

    A->fillComplete();

    Indices.clear();
    Values.clear();

    i = 0;
    for(int l = local_zmin; l <= local_zmax; l++) {
        for(int k = local_ymin; k <= local_ymax; k++) {
            for(int j = local_xmin; j <= local_xmax; j++) {
                double value = u0[ARRAY3D(j,k,l,local_xmin-2, local_ymin-2, local_zmin-2, local_nx, local_ny)];

                b->replaceGlobalValue(myGlobalIndices_[i], value);
                i++;
            }
        }
    }

    i = 0;
    for(int l = local_zmin; l <= local_zmax; l++) {
        for(int k = local_ymin; k <= local_ymax; k++) {
            for(int j = local_xmin; j <= local_xmax; j++) {
                double value = u0[ARRAY3D(j,k,l,local_xmin-2, local_ymin-2, local_zmin-2, local_nx, local_ny)];

                x->replaceGlobalValue(myGlobalIndices_[i], value);
                i++;
            }
        }
    }

    typedef Xpetra::Matrix<Scalar, Ordinal, Ordinal, Node> Xpetra_Matrix;
    typedef Xpetra::CrsMatrix<Scalar, Ordinal, Ordinal, Node> Xpetra_CrsMatrix;
    typedef Xpetra::TpetraCrsMatrix<Scalar, Ordinal, Ordinal, Node> Xpetra_TpetraCrsMatrix;
    typedef Xpetra::CrsMatrixWrap<Scalar, Ordinal, Ordinal, Node> Xpetra_CrsMatrixWrap;
    Teuchos::RCP<Xpetra_CrsMatrix> xcrsA = rcp(new Xpetra_TpetraCrsMatrix(A));
    Teuchos::RCP<Xpetra_Matrix> xopA = rcp(new Xpetra_CrsMatrixWrap(xcrsA));
    Teuchos::RCP<Operator> belosOp = Teuchos::rcp(new Belos::XpetraOp<Scalar,Ordinal,Ordinal,Node>(xopA)); // Turns a Xpetra::Operator object into a Belos operator

    problem = rcp(new Belos::LinearProblem<Scalar, MultiVector, Operator>(belosOp, x, b));
    problem->setOperator(belosOp);
    problem->setLHS(x);
    problem->setRHS(b);

    if(preconditioner_active)
    {
        typedef Belos::MueLuOp<Scalar,Ordinal,Ordinal,Node> Belos_MueLuOperator;

        Teuchos::RCP<MueLu::HierarchyManager<Scalar,Ordinal,Ordinal,Node>>
            mueLuFactory = rcp(new MueLu::ParameterListInterpreter<Scalar,Ordinal,Ordinal,Node>(preconditionerParams));

        Teuchos::RCP<MueLu::Hierarchy<Scalar,Ordinal,Ordinal,Node>> hierarchy;

        hierarchy = mueLuFactory->CreateHierarchy();   
        hierarchy->GetLevel(0)->Set("A"          , xopA        );
        hierarchy->GetLevel(0)->Set("Coordinates", xcoordinates);

        mueLuFactory->SetupHierarchy(*hierarchy);
        //hierarchy->Setup(); // This does the set-up without applying the preconditionerParams settings
        hierarchy->IsPreconditioner(true);
        //Teuchos::ParameterList validParams;
        //validParams = mueLuFactory.GetValidParameterList();
        Teuchos::RCP<Belos_MueLuOperator> preconditioner = rcp(new Belos_MueLuOperator(hierarchy));

        problem->setLeftPrec(preconditioner); //commenting out will disable the MueLu preconditioner 
    }

    problem->setProblem(); //should be called after setting the preconditioner to avoid errors in the CG solver - see issue #2

    solver->setProblem(problem);

    Belos::ReturnType result = solver->solve();

    *numIters = solver->getNumIters();
    if(MyPID == 0)
    {
        std::cout << "[STEM]: num_iters = " << *numIters << std::endl;
    }

    Teuchos::Array<Scalar> solution(numLocalElements_);
    x->get1dCopy(solution, numLocalElements_);

    i = 0;
    for(int l = local_zmin; l <= local_zmax; l++) {
        for(int k = local_ymin; k <= local_ymax; k++) {
            for(int j = local_xmin; j <= local_xmax; j++) {
                u0[ARRAY3D(j,k,l,local_xmin-2, local_ymin-2, local_zmin-2, local_nx, local_ny)] = solution[i];
                i++;
            }
        }
    }
    solver = Teuchos::null;
    //preconditioner = Teuchos::null;
    problem = Teuchos::null;
}

void TrilinosStem::finalise()
{
    Teuchos::TimeMonitor::summarize();

    //Free up the storage - Teuchos::RCP objects just need to have the reference counters decremented to invoke freeing of the object
    //solver = Teuchos::null;
    //preconditioner = Teuchos::null;
    //problem = Teuchos::null;
    b = Teuchos::null;
    x = Teuchos::null;
    A = Teuchos::null;
    map = Teuchos::null;
    xmap = Teuchos::null;
    xcoordinates = Teuchos::null;
    solverParams = Teuchos::null;
    delete myGlobalIndices_;
}

